import pygame

# a panel is the box that flashes
class Panel:
	def __init__(self, rect, freq, depth=0):
		self.rect = pygame.Rect(rect)
		self.freq = freq
		self.interval = 1000 / (freq * 2)
		self.state = True
		self.last_toggle = pygame.time.get_ticks()
		self.active = True
		self.depth = depth

	
	def update(self, now):

		# if it is not activated (another panel was clicked), do nothing (dont update)
		if not self.active:
			return 

		# toggle state based on the interval and the tick
		if now - self.last_toggle >= self.interval:
			self.state = not self.state
			self.last_toggle = now
	
	def draw(self, surface):

		# paint color based on state
		color = "Black" if self.state else "White"

		# if it is inactive, paint gray
		if not self.active:
			color = "Gray"
		pygame.draw.rect(surface, color, self.rect)


	# when a panel inside a grid is clicked or focused (frequency), it becomes a new grid with panels (the recursive subdivision)
	# subdivides based on mouse position or given frequency
	def subdivide(self, pos=None, freq=None):

		# if its inactive, do nothing
		if not self.active:
			return self

		# if the mouse click position is inside the panel
		if pos and self.rect.collidepoint(pos):
			
			# make it a new grid
			new_grid = Grid(self.rect, depth = self.depth + 1)
			print(f"Subdivisions by click {pos}: {self.depth + 1}")
			return new_grid

		# if the frequency position matches the frequency of the panel
		if freq is not None and self.freq == freq:

			# make it a new grid
			new_grid = Grid(self.rect, depth = self.depth + 1)
			print(f"Subdivisions by freq {freq}: {self.depth + 1}")
			return new_grid

		# if nether cases are true, do nothing (stays a panel)
		return self

# a grid contains four panels that flash at different frequencies
class Grid:
	def __init__(self, rect, depth=0, rows=2, cols=2):
		self.active = True
		self.rect = pygame.Rect(rect)
		self.children = []

		# find the size of the child panels dynamically
		w = self.rect.width // cols
		h = self.rect.height // rows

		# the frequencies in hertz of the flashing child panels
		freqs = [15, 20, 25, 30]
		#freqs = [10,16,23,31]

		i = 0
		for r in range(rows):
			for c in range(cols):

				# make children rects
				child_rect = (
					self.rect.x + c * w,
					self.rect.y + r * h,
					w,
					h
				)
				self.children.append(
					Panel(child_rect, freqs[i % len(freqs)], depth=depth)
				)
				i += 1
	
	def update(self, now):

		# if it is not activated (another panel was clicked), do nothing (dont update)
		if not self.active:
			return
		
		# update its children
		for child in self.children:
			child.update(now)


	def draw(self, surface):

		# draw each child
		for child in self.children:
			child.draw(surface)

	# subdivide one of the children of a grid, deactivate the other children
	def subdivide(self, pos=None, freq=None):

		# if the grid is inactive (another child was subdivided), do nothing
		if not self.active:
			return self

		# for each child (keep track of what child with i)
		for i, child in enumerate(self.children):

			# if the child is inactive, skip (do nothing)
			if not child.active:
				continue

			# subdivide the child
			new_child = child.subdivide(pos=pos, freq=freq)

			# check if the child is actaully changed (divided into a grid)
			if new_child is not child:
				self.children[i] = new_child
				new_child.active = True

				# deactivate the other panels
				for j, sibling in enumerate(self.children):
					if j != i:
						sibling.active = False

				break
				
		# returns the grid to itself so the recursive structure remains
		return self


# size of the window
WIDTH = 1500
HEIGHT = 1000


def main():
	
	print(f"Size: {WIDTH} x {HEIGHT}")

	# start the pygame
	pygame.init()
	screen = pygame.display.set_mode((WIDTH, HEIGHT))
	clock = pygame.time.Clock()

	# initiatlize with a grid on the window
	#root = Grid((0, 0, WIDTH, HEIGHT), depth=0)
	root = Panel((0,0,WIDTH,HEIGHT), 31, depth=0)
	running = True

	while running:

		# set time each interation
		now = pygame.time.get_ticks()

		for event in pygame.event.get():
			if event.type == pygame.QUIT:
				running = False

			# if a click occurs, subdivde based on the position
			if event.type == pygame.MOUSEBUTTONDOWN:
				root.subdivide(pos=event.pos)

			# TO DO: subdivide based on the frequency observed from EEG data
			# root.subdivide(freq = observed_freq)

		# make the background of the window the same color as the inactive panels
		screen.fill("Gray")

		# update the root and draw it
		root.update(now)
		root.draw(screen)

		pygame.display.flip()

		# set tick speed to 60 fps (can increase later for higher refresh rate monitors)
		clock.tick(60)

	pygame.quit()




if __name__ == "__main__":
	main()


























